# Etapa de construcción
FROM maven:3.8.6-openjdk-11 AS build
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:go-offline
COPY src ./src
RUN mvn clean package

# Etapa final con imagen JRE moderna
FROM eclipse-temurin:11-jre
WORKDIR /app
COPY --from=build /app/target/segmentacion-web-1.0.jar .
EXPOSE 8080
CMD ["java", "-jar", "segmentacion-web-1.0.jar"]
